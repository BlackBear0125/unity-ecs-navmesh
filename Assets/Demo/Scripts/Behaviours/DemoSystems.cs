#region

using UnityEngine;
using UnityEngine.UI;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using Demo.Behaviours;
using NavJob.Components;
using NavJob.Systems;

#endregion

namespace Demo
{
    public class SpawnSystem : ComponentSystem
    {
        public int pendingSpawn;
        private EntityManager _manager;

        private PopulationSpawner _spawner;
        private int _lastSpawned;
        private float _nextUpdate;

        private Vector3 one = Vector3.one;
        private EntityArchetype agent;

        private int spawned;

        private Text spawnedText;

        private Text SpawnedText
        {
            get
            {
                if (spawnedText == null)
                {
                    spawnedText = GameObject.Find ("SpawnedText").GetComponent<Text> ();
                }

                return spawnedText;
            }
        }

        private PopulationSpawner Getspawner ()
        {
            if (_spawner == null)
            {
                _spawner = Object.FindObjectOfType<PopulationSpawner> ();
            }

            return _spawner;
        }

        private EntityManager Getmanager ()
        {
            if (_manager == null)
            {
                _manager = World.Active.GetOrCreateManager<EntityManager> ();
            }

            return _manager;
        }

        protected override void OnCreateManager (int capacity)
        {
            base.OnCreateManager (capacity);
            // create the system
            World.Active.CreateManager<NavAgentSystem> ();
            agent = Getmanager ().CreateArchetype (
                typeof (NavAgent),
                // optional avoidance
                // typeof(NavAgentAvoidance),
                // optional
                 typeof (Position),
                 typeof (Rotation)
                
            );
        }

        [Inject] private BuildingCacheSystem buildings;
        [Inject] private InjectData data;
        protected override void OnUpdate ()
        {
            if (Time.time > _nextUpdate && _lastSpawned != spawned)
            {
                _nextUpdate = Time.time + 0.5f;
                _lastSpawned = spawned;
                SpawnedText.text = $"Spawned: {spawned} people";
            }

            if (Getspawner ().Renderers.Length == 0)
            {
                return;
            }

            if (buildings.ResidentialBuildings.Length == 0)
            {
                return;
            }

            var spawnData = data.Spawn[0];
            pendingSpawn = spawnData.Quantity;
            spawnData.Quantity = 0;
            data.Spawn[0] = spawnData;
            var manager = Getmanager ();
            for (var i = 0; i < pendingSpawn; i++)
            {
                spawned++;
                var entity = manager.CreateEntity (agent);
                var navAgent = new NavAgent (
                    spawnData.AgentStoppingDistance,
                    spawnData.AgentMoveSpeed,
                    spawnData.AgentAcceleration,
                    spawnData.AgentRotationSpeed,
                    spawnData.AgentAreaMask
                );
                var position = new Position()
                {
                    Value = buildings.GetResidentialBuilding()
                };
                // optional if set on the archetype
                // manager.SetComponentData (entity, new Position { Value = position });
                manager.SetComponentData (entity, navAgent);
                manager.SetComponentData(entity, position);
                // optional for avoidance
                // var navAvoidance = new NavAgentAvoidance(2f);
                // manager.SetComponentData(entity, navAvoidance);
                manager.AddSharedComponentData (entity, Getspawner ().Renderers[UnityEngine.Random.Range (0, Getspawner ().Renderers.Length)].Value);
            }
            return;
        }

        private struct InjectData
        {
            public readonly int Length;
            public ComponentDataArray<PendingSpawn> Spawn;
        }
    }

    public class DetectIdleAgentSystem : ComponentSystem
    {
        public struct AgentData
        {
            public int index;
            public Entity entity;
            public NavAgent agent;
        }

        private Text awaitingNavmeshText;

        private Text AwaitingNavmeshText
        {
            get
            {
                if (awaitingNavmeshText == null)
                {
                    awaitingNavmeshText = GameObject.Find ("AwaitingNavmeshText").GetComponent<Text> ();
                }

                return awaitingNavmeshText;
            }
        }

        private Text cachedPathText;

        private Text CachedPathText
        {
            get
            {
                if (cachedPathText == null)
                {
                    cachedPathText = GameObject.Find ("CachedPathText").GetComponent<Text> ();
                }

                return cachedPathText;
            }
        }

        private float _nextUpdate;

        private NativeQueue<int> needsPath = new NativeQueue<int> (Allocator.Persistent);

        [BurstCompile]
        private struct DetectIdleAgentJob : IJobParallelFor
        {
            public InjectData data;
            public NativeQueue<int>.Concurrent needsPath;

            public void Execute (int index)
            {
                var agent = data.Agents[index];
                if (data.Agents[index].status == AgentStatus.Idle)
                {
                    needsPath.Enqueue (index);
                    agent.status = AgentStatus.PathQueued;
                    data.Agents[index] = agent;
                }
            }
        }

        private struct SetNextPathJob : IJob
        {
            public InjectData data;
            public NativeQueue<int> needsPath;
            public void Execute ()
            {
                while (needsPath.TryDequeue (out int index))
                {
                    var destination = BuildingCacheSystem.GetCommercialBuilding ();
                    NavAgentSystem.SetDestinationStatic (data.Entities[index], data.Agents[index], 
                        data.Positions[index], data.Rotations[index], destination);
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

        [Inject] InjectData data;
        [Inject] NavMeshQuerySystem navQuery;

        protected override void OnUpdate ()
        {
            if (Time.time > _nextUpdate)
            {
                AwaitingNavmeshText.text = $"Awaiting Path: {navQuery.PendingCount} people";
                CachedPathText.text = $"Cached Paths: {navQuery.CachedCount}";
                _nextUpdate = Time.time + 0.5f;
            }
            var inputDeps = new DetectIdleAgentJob { data = data, needsPath = needsPath }.Schedule (data.Length, 64);
            inputDeps = new SetNextPathJob { data = data, needsPath = needsPath }.Schedule (inputDeps);
            inputDeps.Complete ();
        }

        protected override void OnDestroyManager ()
        {
            needsPath.Dispose ();
        }
    }

    public class BuildingCacheSystem : ComponentSystem
    {
        public NativeList<Vector3> CommercialBuildings = new NativeList<Vector3> (Allocator.Persistent);
        public NativeList<Vector3> ResidentialBuildings = new NativeList<Vector3> (Allocator.Persistent);
        private PopulationSpawner spawner;
        private int nextCommercial = 0;
        private int nextResidential = 0;
        private static BuildingCacheSystem instance;

        protected override void OnCreateManager (int capacity)
        {
            instance = this;
        }

        [Inject] private InjectData data;

        private struct InjectData
        {
            public readonly int Length;
            [ReadOnly] public ComponentDataArray<BuildingData> Buildings;
        }

        private PopulationSpawner Spawner
        {
            get
            {
                if (spawner == null)
                {
                    spawner = Object.FindObjectOfType<PopulationSpawner> ();
                }

                return spawner;
            }
        }

        public Vector3 GetResidentialBuilding ()
        {
            nextResidential++;
            if (nextResidential >= ResidentialBuildings.Length)
            {
                nextResidential = 0;
            }

            return ResidentialBuildings[nextResidential];
        }

        public static Vector3 GetCommercialBuilding ()
        {
            var building = instance.CommercialBuildings[0];
            try
            {
                if (instance.nextCommercial < instance.CommercialBuildings.Length)
                {
                    building = instance.CommercialBuildings[instance.nextCommercial];
                    instance.nextCommercial++;
                }
                else
                {
                    instance.nextCommercial = 0;
                }
                return building;
            }
            catch
            {
                return building;
            }
        }

        protected override void OnUpdate ()
        {
            for (var i = 0; i < data.Length; i++)
            {
                var building = data.Buildings[i];
                if (building.Type == BuildingType.Residential)
                {
                    ResidentialBuildings.Add (building.Position);
                }
                else
                {
                    CommercialBuildings.Add (building.Position);
                }

                PostUpdateCommands.RemoveComponent<BuildingData> (building.Entity);
            }
        }

        protected override void OnDestroyManager ()
        {
            ResidentialBuildings.Dispose ();
            CommercialBuildings.Dispose ();
        }
    }
}