#region

using System.Collections.Concurrent;
using System.Collections.Generic;
using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using NavJob.Components;

#endregion

namespace NavJob.Systems
{

    class SetDestinationBarrier : BarrierSystem { }
    class PathSuccessBarrier : BarrierSystem { }
    class PathErrorBarrier : BarrierSystem { }

    [DisableAutoCreation]
    public class NavAgentSystem : JobComponentSystem
    {

        private struct AgentData
        {
            public int index;
            public Entity entity;
            public NavAgent agent;
            public Position position;
            public Rotation rotation;
        }

        private NativeQueue<AgentData> needsWaypoint;
        private ConcurrentDictionary<int, Vector3[]> waypoints = new ConcurrentDictionary<int, Vector3[]> ();
        private NativeHashMap<int, AgentData> pathFindingData;

        [BurstCompile]
        private struct DetectNextWaypointJob : IJobParallelFor
        {
            public int navMeshQuerySystemVersion;
            public InjectData data;
            public NativeQueue<AgentData>.Concurrent needsWaypoint;

            public void Execute (int index)
            {
                var agent = data.Agents[index];
                if (agent.remainingDistance - agent.stoppingDistance > 0 || agent.status != AgentStatus.Moving)
                {
                    return;
                }
                var entity = data.Entities[index];
                if (agent.nextWaypointIndex != agent.totalWaypoints)
                {
                    needsWaypoint.Enqueue (new AgentData
                    {
                        rotation = data.Rotations[index],
                        position =  data.Positions[index],
                        agent = data.Agents[index],
                        entity = entity,
                        index = index
                    });
                }
                else if (navMeshQuerySystemVersion != agent.queryVersion || agent.nextWaypointIndex == agent.totalWaypoints)
                {
                    agent.totalWaypoints = 0;
                    agent.currentWaypoint = 0;
                    agent.status = AgentStatus.Idle;
                    data.Agents[index] = agent;
                }
            }
        }

        private struct SetNextWaypointJob : IJob
        {
            public InjectData data;
            public NativeQueue<AgentData> needsWaypoint;
            public void Execute ()
            {
                while (needsWaypoint.TryDequeue (out AgentData item))
                {
                    var entity = data.Entities[item.index];
                    if (NavAgentSystem.instance.waypoints.TryGetValue (entity.Index, out Vector3[] currentWaypoints))
                    {
                        var agent = data.Agents[item.index];
                        var position = data.Positions[item.index];

                        agent.currentWaypoint = currentWaypoints[agent.nextWaypointIndex];
                        agent.remainingDistance = math.distance(position.Value, agent.currentWaypoint);
                        agent.nextWaypointIndex++;
                        data.Agents[item.index] = agent;
                    }
                }
            }
        }

        [BurstCompile]
        private struct MovementJob : IJobParallelFor
        {
            private readonly float dt;
            private readonly float3 up;
            private readonly float3 one;

            private InjectData data;

            public MovementJob (InjectData data, float dt)
            {
                this.dt = dt;
                this.data = data;
                up = Vector3.up;
                one = Vector3.one;
            }

            public void Execute (int index)
            {
                if (index >= data.Agents.Length)
                {
                    return;
                }

                
                var agent = data.Agents[index];
                var position = data.Positions[index];
                var rotation = data.Rotations[index];

                if (agent.status != AgentStatus.Moving)
                {
                    return;
                }

                if (agent.remainingDistance > 0)
                {
                    agent.currentMoveSpeed = Mathf.Lerp (agent.currentMoveSpeed, agent.moveSpeed, dt * agent.acceleration);
                    // todo: deceleration
                    if (agent.nextPosition.x != Mathf.Infinity)
                    {
                        position.Value = agent.nextPosition;
                    }
                    var heading = agent.currentWaypoint - position.Value;
                    agent.remainingDistance = math.length(heading);
                    if (agent.remainingDistance > 0.001f)
                    {
                        heading.y = 0.0f;
                        heading = math.normalize(heading);
                        var targetRotation = quaternion.lookRotation(heading, up);
                        if (agent.remainingDistance < 1)
                        {
                            rotation.Value = targetRotation;
                        }
                        else
                        {
                            rotation.Value = math.slerp(rotation.Value, targetRotation, dt * agent.rotationSpeed);
                        }
                    }
                    var forward = math.forward (rotation.Value) * agent.currentMoveSpeed * dt;
                    agent.nextPosition = position.Value + forward;
                    data.Agents[index] = agent;
                    data.Positions[index] = position;
                    data.Rotations[index] = rotation;
                }
                else if (agent.nextWaypointIndex == agent.totalWaypoints)
                {
                    agent.nextPosition = new float3 { x = Mathf.Infinity, y = Mathf.Infinity, z = Mathf.Infinity };
                    agent.status = AgentStatus.Idle;
                    data.Agents[index] = agent;
                }
            }
        }

        private struct InjectData
        {
            public readonly int Length;
            [ReadOnly] public EntityArray Entities;
            public ComponentDataArray<NavAgent> Agents;
            public ComponentDataArray<Position> Positions;
            public ComponentDataArray<Rotation> Rotations;
        }

        [Inject] private InjectData data;
        [Inject] private NavMeshQuerySystem querySystem;
        [Inject] SetDestinationBarrier setDestinationBarrier;
        [Inject] PathSuccessBarrier pathSuccessBarrier;
        [Inject] PathErrorBarrier pathErrorBarrier;

        protected override JobHandle OnUpdate (JobHandle inputDeps)
        {
            var dt = Time.deltaTime;
            inputDeps = new DetectNextWaypointJob { data = data, needsWaypoint = needsWaypoint, navMeshQuerySystemVersion = querySystem.Version }.Schedule (data.Length, 64, inputDeps);
            inputDeps = new SetNextWaypointJob { data = data, needsWaypoint = needsWaypoint }.Schedule (inputDeps);
            inputDeps = new MovementJob (data, dt).Schedule (data.Length, 64, inputDeps);
            return inputDeps;
        }

        /// <summary>
        /// Used to set an agent destination and start the pathfinding process
        /// </summary>
        /// <param name="entity"></param>
        /// <param name="agent"></param>
        /// <param name="destination"></param>
        public void SetDestination (Entity entity, NavAgent agent, Position position, Rotation rotation, float3 destination)
        {
            if (pathFindingData.TryAdd (entity.Index, new AgentData { index = entity.Index, entity = entity, agent = agent, position = position, rotation = rotation}))
            {
                var command = setDestinationBarrier.CreateCommandBuffer ();
                agent.status = AgentStatus.PathQueued;
                agent.destination = destination;
                agent.queryVersion = querySystem.Version;
                command.SetComponent<NavAgent>(entity, agent);
                querySystem.RequestPath (entity.Index, position.Value, agent.destination, agent.areaMask);
            }
        }

        /// <summary>
        /// Static counterpart of SetDestination
        /// </summary>
        /// <param name="entity"></param>
        /// <param name="agent"></param>
        /// <param name="destination"></param>
        public static void SetDestinationStatic (Entity entity, NavAgent agent, Position position, Rotation rotation, float3 destination)
        {
            instance.SetDestination (entity, agent, position, rotation, destination);
        }

        protected static NavAgentSystem instance;

        protected override void OnCreateManager (int capacity)
        {
            instance = this;
            querySystem.RegisterPathResolvedCallback (OnPathSuccess);
            querySystem.RegisterPathFailedCallback (OnPathError);
            needsWaypoint = new NativeQueue<AgentData> (Allocator.Persistent);
            pathFindingData = new NativeHashMap<int, AgentData> (0, Allocator.Persistent);
        }

        protected override void OnDestroyManager ()
        {
            needsWaypoint.Dispose ();
            pathFindingData.Dispose ();
        }

        private void SetWaypoint (Entity entity, NavAgent agent, Position position, Rotation rotation, Vector3[] newWaypoints)
        {
            waypoints[entity.Index] = newWaypoints;
            var command = pathSuccessBarrier.CreateCommandBuffer ();
            agent.status = AgentStatus.Moving;
            agent.nextWaypointIndex = 1;
            agent.totalWaypoints = newWaypoints.Length;
            agent.currentWaypoint = newWaypoints[0];
            agent.remainingDistance = math.distance(position.Value, agent.currentWaypoint);
            command.SetComponent<NavAgent> (entity, agent);
            command.SetComponent<Position>(entity, position);
            command.SetComponent<Rotation>(entity, rotation);
        }

        private void OnPathSuccess (int index, Vector3[] waypoints)
        {
            if (pathFindingData.TryGetValue (index, out AgentData entry))
            {
                SetWaypoint (entry.entity, entry.agent, entry.position, entry.rotation, waypoints);
                pathFindingData.Remove (index);
            }
        }

        private void OnPathError (int index, PathfindingFailedReason reason)
        {
            if (pathFindingData.TryGetValue (index, out AgentData entry))
            {
                entry.agent.status = AgentStatus.Idle;
                var command = pathErrorBarrier.CreateCommandBuffer ();
                command.SetComponent<NavAgent>(entry.entity, entry.agent);
                command.SetComponent<Position>(entry.entity, entry.position);
                command.SetComponent<Rotation>(entry.entity, entry.rotation);
                pathFindingData.Remove (index);
            }
        }
    }
}